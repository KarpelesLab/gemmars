<script setup>
// A model reply that grows token by token. Each new piece of text lands in copper and oxidizes to
// ink, so the reply visibly patinates as it streams. Earlier text is never re-rendered unless the
// tokenizer revises it (rare: a multi-byte character completing), in which case the reply resets.
import { ref, watch } from 'vue';

const props = defineProps({ text: { type: String, required: true } });
const pieces = ref([]);
let shown = '';
let id = 0;

watch(
  () => props.text,
  (t) => {
    if (t.startsWith(shown)) {
      const add = t.slice(shown.length);
      if (add) pieces.value.push({ id: id++, text: add });
    } else {
      pieces.value = [{ id: id++, text: t }];
    }
    shown = t;
  },
  { immediate: true },
);
</script>

<template>
  <p class="reply"><span v-for="p in pieces" :key="p.id" class="piece">{{ p.text }}</span></p>
</template>

<style scoped>
.reply {
  margin: 0;
  white-space: pre-wrap;
  overflow-wrap: anywhere;
}
.piece {
  animation: oxidize 1.4s ease-out both;
}
@keyframes oxidize {
  from {
    color: var(--copper);
  }
  to {
    color: var(--ink);
  }
}
@media (prefers-reduced-motion: reduce) {
  .piece {
    animation: none;
  }
}
</style>
